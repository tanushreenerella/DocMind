"use client";

import Image from "next/image";
import { FileText } from "lucide-react";
import { getPageImageUrl } from "@/lib/api";

interface CitationCardProps {
  docName: string;
  pageNumber: number;
  imageFilename: string;
  excerpt: string;
  onExpand: (imageUrl: string, docName: string, pageNumber: number) => void;
}

export default function CitationCard({
  docName,
  pageNumber,
  imageFilename,
  excerpt,
  onExpand,
}: CitationCardProps) {
  const imageUrl = imageFilename ? getPageImageUrl(imageFilename) : "";

  return (
    <div className="flex gap-3 p-3 bg-gray-50 border border-gray-200 rounded-xl hover:border-indigo-200 hover:bg-indigo-50/30 transition-all duration-200">
      {/* Thumbnail — only clickable when a page image exists */}
      <div
        onClick={imageUrl ? () => onExpand(imageUrl, docName, pageNumber) : undefined}
        className={`shrink-0 w-16 h-20 rounded-lg overflow-hidden border border-gray-200 bg-white flex items-center justify-center ${imageUrl ? "cursor-pointer hover:scale-105 hover:shadow-md transition-all duration-200" : ""}`}
        title={imageUrl ? "View full page" : undefined}
      >
        {imageUrl ? (
          <div className="relative w-full h-full">
            <Image
              src={imageUrl}
              alt={`${docName} page ${pageNumber}`}
              fill
              className="object-cover"
              unoptimized
              onError={(e) => {
                (e.target as HTMLImageElement).style.display = "none";
              }}
            />
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100">
              <FileText className="w-6 h-6 text-gray-400" />
            </div>
          </div>
        ) : (
          <FileText className="w-6 h-6 text-gray-400" />
        )}
      </div>

      {/* Info */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-semibold text-gray-800 truncate">{docName}</p>
        <p className="text-xs text-indigo-600 font-medium mt-0.5">Page {pageNumber}</p>
        <p className="text-xs text-gray-500 mt-1.5 line-clamp-3 leading-relaxed">
          {excerpt}
        </p>
      </div>
    </div>
  );
}
